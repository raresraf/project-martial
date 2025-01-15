#include <iostream>
#include <string>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n;
  cin >> n;

  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    int m = s.length();
    if (m > 10) {
      cout << s[0] + to_string(m - 2) + s[m - 1] << "\n";
    } else {
      cout << s << "\n";
    }
  }
  return 0;
}