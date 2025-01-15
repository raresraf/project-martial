#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, k;
  string s;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> s;
    k = s.size();
    if (k > 10)
      cout << s[0] << k - 2 << s[k - 1];
    else
      cout << s;
    cout << endl;
  }
  return 0;
}
