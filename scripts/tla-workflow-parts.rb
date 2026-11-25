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
      abort "Unsupported conditional or compound TLA invocation: #{line}" if step.key?('if') || words.length > 1
      if words == ['__matrix_part__']
        puts job.fetch('strategy').fetch('matrix').fetch('part')
      else
        puts words.first || 'all'
      end
    end
  end
end
