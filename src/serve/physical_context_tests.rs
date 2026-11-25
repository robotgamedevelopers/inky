#[test]
fn physical_context_clamps_both_apis_before_they_start_streaming() {
    for (full_slots, window_slots) in [(1, 64), (64, 1)] {
        for messages_api in [false, true] {
            for stream in [false, true] {
                for (prompt_len, expected_budget) in [(60, Some(5)), (64, Some(1)), (65, None)] {
                    let (mut cx, jobs) = admission_test_ctx(1);
                    cx.ctx = pooled_context(4096, full_slots, window_slots);
                    let mut request = json!({
                        "messages": [{"role": "user", "content": "x"}],
                        "max_tokens": 128000,
                        "loopcd_omega": 0.5,
                        "stream": stream,
                    });
                    let rendered_len = |req: &Value| {
                        if messages_api {
                            render_messages(&cx, req, usize::MAX).unwrap().len()
                        } else {
                            render(&cx.tk, &cx.sp, req, &cx.effort).unwrap().len()
                        }
                    };
                    // The fixture emits one token per ASCII byte. Include the actual
                    // endpoint's template when positioning the last prompt token.
                    let padding = prompt_len - rendered_len(&request);
                    request["messages"][0]["content"] = json!("x".repeat(1 + padding));
                    assert_eq!(rendered_len(&request), prompt_len);
                    let eos = cx.tk.eos;
                    let scheduler = std::thread::spawn(move || {
                        let job = jobs.recv().ok()?;
                        let observed = (job.toks.len(), job.max, job.loopcd_omega);
                        job.out.send(Ev::Tok(eos)).unwrap();
                        job.out
                            .send(Ev::Done {
                                prompt: job.toks.len(),
                                cached: 0,
                                gen: 1,
                                reason: "stop",
                                timing: Default::default(),
                            })
                            .unwrap();
                        Some(observed)
                    });
                    let mut out = Vec::new();
                    if messages_api {
                        messages(&cx, &request, &mut out).unwrap();
                    } else {
                        chat(&cx, &request, &mut out).unwrap();
                    }
                    drop(cx); // Wake the mock scheduler when validation rejected the request.
                    let submitted = scheduler.join().unwrap();
                    let response = String::from_utf8(out).unwrap();
                    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
                    if let Some(budget) = expected_budget {
                        assert_eq!(submitted, Some((prompt_len, budget, 0.5)));
                        assert!(headers.starts_with("HTTP/1.1 200 "), "{headers}");
                        assert_eq!(headers.contains("text/event-stream"), stream);
                    } else {
                        assert_eq!(submitted, None, "oversized request reached the scheduler");
                        assert!(headers.starts_with("HTTP/1.1 400 "), "{headers}");
                        assert!(headers.contains("application/json"), "{headers}");
                        assert!(
                            serde_json::from_str::<Value>(body).unwrap()["error"]["message"]
                                .is_string()
                        );
                    }
                }
            }
        }
    }
}
