#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string input;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> input;
    if (input.size() > 10) {
      int temp = 0;
      for (int i = 1; i < input.size() - 1; i++) {
        temp++;
      }
      cout << input[0] << temp << input[input.size() - 1] << endl;
    } else {
      cout << input << endl;
    }
  }

  return 0;
}
