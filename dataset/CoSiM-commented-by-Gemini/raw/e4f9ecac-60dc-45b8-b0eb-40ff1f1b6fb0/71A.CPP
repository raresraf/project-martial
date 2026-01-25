// CodeForces
// Problem 71A [Way Too Long Words]

#include <iostream>
#include <string>
using namespace std;

int main() {
  string input;

  unsigned int test_cases = 0;
  cin >> test_cases;

  while (test_cases--) {
    cin >> input;

    if (input.length() > 10) {
      cout << input[0] << input.length() - 2 << input[input.length() - 1]
           << endl;
    } else {
      cout << input << endl;
    }
  }
  return 0;
}