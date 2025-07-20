#include <iostream>
using namespace std;

#define end return 0

typedef long long ll;
typedef long long int lli;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(0);
  lli a, b, c, d, e;
  cin >> a >> b;
  string s;
  cin >> s;
  while (b--) {
    for (int i = 0; i < a - 1; ++i) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i++;
      }
    }
  }
  cout << s << endl;
  return 0;
}