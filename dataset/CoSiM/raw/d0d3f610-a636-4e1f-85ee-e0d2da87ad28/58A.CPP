#include <bits/stdc++.h>
using namespace std;

int main(void) {

  // freopen("c.in" , "r" , stdin);

  string s;
  cin >> s;
  int counter = 0;
  string hello = "hello";

  for (int i = 0, j = s.length(); i < j; i++) {
    if (s[i] == hello[counter]) {
      counter++;
    }
  }
  if (counter == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
