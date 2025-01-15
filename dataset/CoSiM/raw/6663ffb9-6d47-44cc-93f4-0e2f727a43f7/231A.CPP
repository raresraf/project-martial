// By archit-dev, contest: Codeforces Round #143 (Div. 2), problem: (A) Team,
// Accepted, #, Copy
#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, petya, vatsya, tonya, count = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> petya >> vatsya >> tonya;
    if (petya + vatsya + tonya >= 2) {
      count++;
    } else
      continue;
  }
  cout << count << "\n";
  return 0;
}
