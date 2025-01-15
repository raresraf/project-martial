#include <iostream>
#include <string>

using namespace std;

#define endl '\n'

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  int n, l;
  cin >> n;
  cin.get(); // returns newline \n

  string s;
  for (int i = 0; i < n; ++i) {
    getline(cin, s);
    l = s.length();
    if (l > 10) {
      cout << s[0] << l - 2 << s[l - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
}