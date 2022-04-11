#ifndef SUDOKU4ANDROID_CLI_OPTS_H
#define SUDOKU4ANDROID_CLI_OPTS_H

#include <string>
#include <vector>

/**
 * Simple command line option parser based on std::find.
 */
class CliOptionsParser
{
    std::vector<std::string> args;

  public:
    explicit CliOptionsParser(int argc, char** argv)
    {
        for (int i = 1; i < argc; ++i) {
            this->args.emplace_back(argv[i]);
        }
    }

    /**
     * Returns the string value following the given option. If the option is not present an empty string is returned.
     * If the same option is specified multiple times, the first value is returned.
     *
     * @param option Name of the option
     * @return Option value or empty string
     */
    [[nodiscard]] auto getOption(const std::string_view& option) const -> const std::string&
    {
        auto optIter = std::find(this->args.begin(), this->args.end(), option);
        if (optIter != this->args.end() && ++optIter != this->args.end()) {
            return *optIter;
        }

        // Return reference to empty string. We cannot return a reference to an empty literal as this would result in
        // a reference to a temporary object, i.e. the std::string created.
        static const std::string noOptValue;
        return noOptValue;
    }

    /**
     * Returns the positional value at the given index. If the index is out of bounds an empty string is returned.
     *
     * @param idx Value index. 0 is the first option, not the command name
     * @return Value or empty string
     */
    [[nodiscard]] auto getPositionalOption(size_t idx) const -> const std::string&
    {
        if (idx < args.size()) {
            return args.at(idx);
        }

        static const std::string noOptValue;
        return noOptValue;
    }

    /**
     * Checks werther the given option is present. This does not ensure, that the options actually has a value and
     * should only be used to check for flag-like options.
     * @param option Name of the option
     * @return True if the option is present, false otherwise
     */
    [[nodiscard]] auto hasOption(const std::string_view& option) const -> bool
    {
        return std::find(this->args.begin(), this->args.end(), option) != this->args.end();
    }
};

#endif // SUDOKU4ANDROID_CLI_OPTS_H
