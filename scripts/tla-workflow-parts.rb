# Print the tier selections made by actual check-tla.sh commands within each job.
require 'yaml'
require 'shellwords'

YAML.safe_load(File.read(ARGV.fetch(0))).fetch('jobs').each_value do |job|
  job.fetch('steps', []).each do |step|
    step.fetch('run', '').each_line do |line|
      line = line.strip
      next if line.empty? || line.start_with?('#') || !line.include?('scripts/check-tla.sh')
      words = Shellwords.split(line.gsub(/\$\{\{\s*matrix\.part\s*\}\}/, '__matrix_part__'))
      words.shift while words.first && words.first.match?(/\A\w+=/)
      next unless %w[scripts/check-tla.sh ./scripts/check-tla.sh].include?(words.shift)
      if job.key?('if') || step.key?('if') || step['continue-on-error'] || job['continue-on-error']
        abort 'TLA tier coverage requires an unconditional, required runner'
      end
      commands = step['run'].lines.map(&:strip).reject { |cmd| cmd.empty? || cmd.start_with?('#') }
      abort "Unsupported compound TLA invocation: #{line}" if commands.length != 1 || words.length > 1
      if words == ['__matrix_part__']
        matrix = job.fetch('strategy').fetch('matrix')
        abort 'TLA tier coverage requires one explicit part matrix' unless matrix.keys == ['part'] && matrix['part'].is_a?(Array)
        puts matrix['part']
      else
        puts words.first || 'all'
      end
    end
  end
end
