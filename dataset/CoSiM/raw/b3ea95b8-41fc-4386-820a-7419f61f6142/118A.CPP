#include <algorithm>
#include <cstring>
#include <ctype.h>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(0);
  string a, ss;
  cin >> a;
  vector<char> v;
  ss = a.size();
  for (int i = 0; i < a.size(); ++i) {
    a[i] = tolower(a[i]);
    // dumb y is not a vowel or what? :(
    if (a[i] != 'a' && a[i] != 'e' && a[i] != 'i' && a[i] != 'o' &&
        a[i] != 'u' && a[i] != 'y') {
      v.push_back('.');
      v.push_back((a[i]));
    }
  }
  for (int i = 0; i < v.size(); ++i) {
    cout << v[i];
  }
  return 0;
}
