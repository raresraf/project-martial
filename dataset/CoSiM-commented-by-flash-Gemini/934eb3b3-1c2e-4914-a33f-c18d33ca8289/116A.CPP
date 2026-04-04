/**
 * @file 116A.CPP
 * @brief Calculates the maximum capacity required for a tram.
 *
 * This program simulates a tram journey with 'n' stops. At each stop, it records
 * the number of passengers who exit and the number who enter. The objective is
 * to determine the maximum number of passengers present on the tram at any point
 * throughout its journey. This version handles the running total and maximum
 * capacity calculation.
 *
 * Algorithm: Iterative simulation. Keep a running total of passengers and update
 *            the maximum observed value in each step.
 * Time Complexity: O(N), where N is the number of stops, as each stop is processed once.
 * Space Complexity: O(1).
 */
/// Jos  Luiz da Silva Neto
/// Computer Engineering - Federal University of Itajub
#include <bits/stdc++.h> // Includes most standard library headers.

using namespace std; // Uses the standard namespace.

#define INF 0x3F3F3F3F         // Functional Utility: Defines a large integer value for infinity.
#define LINF 0x3F3F3F3F3F3F3F3FLL // Functional Utility: Defines a large long long integer value for infinity.
#define pb push_back           // Functional Utility: Defines 'pb' as a shorthand for 'push_back'.
#define mp make_pair           // Functional Utility: Defines 'mp' as a shorthand for 'make_pair'.
#define F first                // Functional Utility: Defines 'F' as a shorthand for 'first' for pairs.
#define S second               // Functional Utility: Defines 'S' as a shorthand for 'second' for pairs.

typedef pair<int, int> pii;    // Functional Utility: Defines 'pii' as a shorthand for 'pair<int, int>'.
typedef vector<int> vi;        // Functional Utility: Defines 'vi' as a shorthand for 'vector<int>'.
typedef vector<pii> vii;       // Functional Utility: Defines 'vii' as a shorthand for 'vector<pair<int, int>>'.
typedef long long ll;          // Functional Utility: Defines 'll' as a shorthand for 'long long'.
typedef unsigned long long ull; // Functional Utility: Defines 'ull' as a shorthand for 'unsigned long long'.

#define D(x) cout << #x << " = " << x << endl; // Functional Utility: Debug macro to print variable name and value.

// Declared but unused arrays, possibly for another problem context or general template.
int vx[] = {1, 0, -1, 0};
int vy[] = {0, 1, 0, -1};

int main() {
  /**
   * @brief Main function: Simulates tram journey to find maximum passenger count.
   * @return 0 if the program executes successfully.
   */
  int n; // Number of stops.
  int resp = 0, tam = 0; // 'resp': maximum observed passengers, 'tam': current passengers on tram.
  cin >> n; // Read the number of stops.
  /**
   * @brief Loop through each stop of the tram journey.
   * @pre 'n' is a positive integer. 'resp' and 'tam' are initialized to 0.
   * @post 'resp' holds the maximum number of people on the tram throughout the journey.
   * @invariant 'tam' accurately reflects the number of people on the tram after each stop.
   */
  for (int i = 0; i < n; i++) {
    int x, y; // 'x': passengers leaving, 'y': passengers entering.
    cin >> x >> y; // Read the number of passengers leaving and entering.
    tam = tam - x + y; // Update the current number of people on the tram.
    resp = max(resp, tam); // Functional Utility: Update 'resp' with the maximum value seen so far.
    tam = max(tam, 0); // Invariant: Ensure 'tam' (current passengers) never drops below zero.
                       // This implicitly handles cases where more people leave than are on the tram,
                       // effectively simulating the tram being empty in such cases.
  }
  printf("%d\n", resp); // Output the maximum number of people observed on the tram.
  return 0; // Return 0 for successful execution.
}