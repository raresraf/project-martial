#include <iostream>

using namespace std;

#define long long ll;

int k, n, w;
void solve() {
  int totalprice = 0;
  for (int i = 1; i <= w; i++) {
    totalprice += i * k;
  }
  int solution = n - totalprice;
  if (solution > 0)
    solution = 0;
  cout << abs(solution);
}

int main() {
  cin >> k >> n >> w;
  solve();
}