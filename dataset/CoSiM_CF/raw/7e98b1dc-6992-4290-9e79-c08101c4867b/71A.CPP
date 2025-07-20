#include <cstdio>
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int tc;
  scanf("%d", &tc);
  while (tc--) {
    string s;
    cin >> s;
    if (s.size() > 10) {
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }

  return 0;
}
