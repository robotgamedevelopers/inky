use crate::scheduler::{ServeLaneInput, ServeStepPlan, ServeStepPlanner};

pub(super) const SERVE_STEP_MAX_BYPASS: u32 = 16;

/// The live scheduler calls this after it has applied backpressure and cache deferrals.
#[inline]
pub(super) fn plan(
    planner: &mut ServeStepPlanner,
    inputs: &[ServeLaneInput],
    budget: usize,
    align: usize,
) -> Option<ServeStepPlan> {
    #[cfg(test)]
    let before = super::scheduler_trace::snapshot(planner, inputs.len());
    let decision = planner.plan_and_begin_step(inputs, budget, align);
    #[cfg(test)]
    if let Some(ages) = before {
        super::scheduler_trace::record(planner, inputs, budget, align, &ages, decision.as_ref());
    }
    decision
}
