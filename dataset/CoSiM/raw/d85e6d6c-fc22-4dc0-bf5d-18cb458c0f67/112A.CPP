#include <bits/stdc++.h>
using namespace std;
int main(int argc, char const *argv[]) {
  /* code */
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  string s1, s2;
  cin >> s1;
  cin >> s2;

  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  if (s1 == s2) {
    cout << "0";
  }

  else if (s1 < s2) {
    cout << "-1";

  }

  else {
    cout << "1";
  }
  return 0;
}
