#include <iostream>

using namespace std;

int main() {
  int n, a, b, cap = 0, maxi = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    cap -= a;
    cap += b;
    if (cap > maxi)
      maxi = cap;
  }
  cout << maxi;
  return 0;
}