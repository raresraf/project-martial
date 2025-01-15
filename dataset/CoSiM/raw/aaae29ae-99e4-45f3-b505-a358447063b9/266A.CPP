#include <iostream>

using namespace std;

int main() {

  int n, reps = 1, max = 1, total = 0;
  string s;
  char a, b;

  cin >> n;

  cin >> s;

  a = s[0];

  for (int i = 1; i < n; ++i) {
    b = s[i];

    if (a == b) {
      reps++;
      if (max < reps) {
        max = reps;
      }
      if (i == n - 1) {
        total += max - 1;
      }
    } else if (reps != 1) {
      total += max - 1;
      reps = 1;
      max = 1;
    }
    a = b;
  }

  cout << total << endl;

  return 0;
}