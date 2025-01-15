#include <iostream>
using namespace std;

int main() {

  int n;
  cin >> n;
  string x[n];
  for (int i = 0; i < n; i++) {
    cin >> x[i];
  }

  for (int i = 0; i < n; i++) {
    string s = x[i];
    if (s.length() > 10)
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << "\n";
    else
      cout << s << "\n";
  }
}
