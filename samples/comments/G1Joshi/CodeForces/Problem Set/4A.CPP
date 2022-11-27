#include <bits/stdc++.h>
using namespace std;

int water(int n) {
  for (int i = 2; i <= sqrt(n); i = i + 2) {
    if (!((n - i) % 2)) {
      return 1;
    }
  }
  return 0;
}

int main() {
  int n;
  cin >> n;
  if (!(n % 2)) {
    if (water(n))
      cout << "YES";
    else
      cout << "NO";
  } else
    cout << "NO";
  return 0;
}
