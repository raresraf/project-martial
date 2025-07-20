#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;

  transform(s.begin(), s.end(), s.begin(), ::tolower);

  for (char ch : s) {
    if (ch != 'a' && ch != 'y' && ch != 'u' && ch != 'o' && ch != 'i' &&
        ch != 'e')
      cout << "." << ch;
  }
  cout << endl;
}

/* output -

tour
.t.r

Codeforces
.c.d.f.r.c.s

aBAcAba
.b.c.b

*/
