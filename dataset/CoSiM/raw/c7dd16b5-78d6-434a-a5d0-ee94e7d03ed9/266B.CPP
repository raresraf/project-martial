#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main() {
  string s, a;
  int ss, t;
  cin >> ss >> t >> s;
  // cout<<ss<<" "<<t<<" "<<s;
  for (int i = 0; i < t; i++) {
    for (int j = 0; j < ss - 1; j++) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        s[j] = 'G';
        s[j + 1] = 'B';
        j++;
      }
    }
  }
  cout << s;
  return 0;
}
