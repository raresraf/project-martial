#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;

  int val = 0;

  string str;

  for (int i = 0; i < n; i++) {
    cin.ignore();
    getline(cin, str);

    for (char ch : str) {
      if (ch == '-') {
        val--;
        break;
      }
      if (ch == '+') {
        val++;
        break;
      }
    }
  }

  cout << val;
}