#include <iostream>
// #include <algorithm>
// #include <vector>
#include <set>
// #include <map>

// #define lli long long int

using namespace std;

int main() {
  int n, l;
  string s;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> s;
    l = s.length();
    if (l > 10) {
      cout << s[0] << l - 2 << s[l - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
}
