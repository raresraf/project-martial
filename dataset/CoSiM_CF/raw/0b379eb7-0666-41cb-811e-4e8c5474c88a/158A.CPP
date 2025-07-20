#include <iostream>

using namespace std;

int main() {
  int n, k, num = 0;
  int scores[100];
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> scores[i];
  }
  for (int i = 0; i < n; i++) {
    if (scores[i] >= scores[k - 1] && scores[i] > 0) {
      num++;
    }
  }
  cout << num << "\n";
  return 0;
}