/**
 * @file 231A.CPP
 * @brief Solution for CodeForces Problem 231A - "Team".
 *
 * @details
 * The problem is to count the number of problems a team will implement.
 * The team decides to solve a problem if at least two of the three members are sure.
 *
 * @algorithm Line-by-line Summation
 * This approach is the most efficient in terms of both time and memory.
 * 1. Read the total number of problems, `n`.
 * 2. Loop `n` times.
 * 3. In each iteration, read the three opinions (0 or 1) for a single problem.
 * 4. Sum the three opinions.
 * 5. If the sum is 2 or greater, increment a counter.
 * 6. After the loop, print the final counter.
 *
 * @time_complexity O(N), where N is the number of problems.
 * @space_complexity O(1), as each problem's data is processed without being stored.
 */
#include <iostream>
using namespace std;

int main() {
    int n; // The total number of problems.
    cin >> n;

    int count = 0; // Counter for the number of problems to be implemented.

    /**
     * @brief Loop to process each problem one by one.
     */
    while (n--) {
        int a, b, c; // Opinions of the three team members.
        cin >> a >> b >> c;

        // Sum the opinions. If the sum is 2 or 3, the team is confident enough.
        if (a + b + c >= 2) {
            count++;
        }
    }

    cout << count << endl;
    return 0;
}
