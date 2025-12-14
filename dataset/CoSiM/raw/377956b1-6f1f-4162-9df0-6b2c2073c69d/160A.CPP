#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int cho[n];
  for (int i = 0; i < n; i++) {
    cin >> cho[i];
  }

  sort(cho, cho + n);
  int f = 0;
  int F = 0;
  for (int i = n - 1; i >= 0; i--) {
    f += cho[i];
    int sum = 0;
    for (int j = i - 1; j >= 0; j--) {
      sum += cho[j];
    }
    F++;
    if (f > sum) {
      cout << F << endl;
      break;
    }
  }

  return 0;
}
