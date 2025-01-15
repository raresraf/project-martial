#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int input, advance = 0, temp;
  for (int i = 0; i < n; i++) {
    cin >> input;
    if (i < k - 1 && input > 0) {
      advance++;
    } else if (i + 1 == k && input > 0) {
      temp = input;
      advance++;
    } else if (input >= temp) {
      advance++;
    }
  }
  cout << advance;

  return 0;
}
