#include <iostream>

using namespace std;

#define long long ll;

int n, arr[1000][3];

void solve() {
  int answer = 0;
  for (int *i : arr) {
    if (i[0] + i[1] + i[2] > 1)
      answer++;
  }
  cout << answer;
}

int main() {
  cin >> n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }
  solve();
}