#include <bits/stdc++.h>
#include <string>
using namespace std;
int main() {
  int n, x = 0;
  cin >> n;
  string str;
  // fflush(stdin);
  for (int i = 0; i < n; i++) {
    cin >> str;
    // fflush(stdin);
    if (str[0] == '+' || str[1] == '+') {
      x++;
    } else if (str[0] == '-' || str[1] == '-') {
      x--;
    }
  }
  cout << x << endl;
}
