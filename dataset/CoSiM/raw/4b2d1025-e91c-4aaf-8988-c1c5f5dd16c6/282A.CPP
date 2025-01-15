#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, x = 0;
  string p;
  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> p;

    if (p == "X++" || p == "++X")
      ++x;

    if (p == "X--" || p == "--X")
      --x;
  }

  cout << x;
}
