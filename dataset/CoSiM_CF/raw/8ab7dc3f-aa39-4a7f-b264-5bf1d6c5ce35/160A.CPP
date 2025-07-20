#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, sum = 0;
  cin >> n;
  int numbers[n];
  for (int i = 0; i < n; i++) {
    cin >> numbers[i];
    sum = sum + numbers[i];
  }
  sort(numbers, numbers + n);
  int div = sum / 2;
  int sum2 = 0;
  int count = 0;
  for (int i = n - 1; i >= 0; i--) {
    if (numbers[i] > div)
      cout << 1 << endl;
    else {
      sum2 = sum2 + numbers[i];
      count++;
      if (sum2 > div) {
        cout << count << endl;
        break;
      }
    }
  }

  return 0;
}
