#include <iostream>

using namespace std;

#define long long ll;

int n, k, arr[100];

void solve() {
  int answer = 0;
  for (int i : arr) {
    if (arr[k - 1] <= i && i > 0)
      answer++;
  }
  cout << answer;
}

int main() {
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  solve();
}