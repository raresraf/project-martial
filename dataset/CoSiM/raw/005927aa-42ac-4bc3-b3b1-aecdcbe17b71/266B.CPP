#include <cstdio>
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  while (cin >> n) {
    int t, len;
    string s;
    cin >> t >> s;
    len = s.size();
    while (t--) {

      for (int i = 0; i < len - 1; i++) {
        if (s[i] == 'B' && s[i + 1] == 'G') {
          s[i] = 'G';
          s[i + 1] = 'B';
          i++;
        }
      }
      // cout<<t<<" "<<s<<endl;
    }
    cout << s << endl;
  }

  return 0;
}
