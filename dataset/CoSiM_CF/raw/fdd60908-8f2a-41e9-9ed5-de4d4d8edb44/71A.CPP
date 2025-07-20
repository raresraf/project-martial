#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int t;
  cin >> t;
  while (t > 0) {
    cin >> s;
    int n = s.size();
    if (n > 10)
      cout << s[0] + to_string(n - 2) + s[n - 1] << endl;
    else
      cout << s << endl;
    t--;
  }
}