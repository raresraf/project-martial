#include <iostream>
#include <string>
#define endl "\n"
using namespace std;
int main() {
  int n, i, cnt(0);
  string s;
  cin >> n >> s;
  for (i = 0; i < n - 1; i++) {
    if (s[i] == s[i + 1]) {
      cnt++;
      // i++;
    }
  }
  cout << cnt << endl;
  return 0;
}
