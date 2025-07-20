#include <bits/stdc++.h>
using namespace std;

#define DBG(x) cout << #x << " = " << (x) << "\n"
#define END(e)                                                                 \
  {                                                                            \
    cout << (e) << "\n";                                                       \
    return EXIT_SUCCESS;                                                       \
  }

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  string s1;
  char c;
  cin >> s1;
  for (unsigned int i = 0; i < s1.size(); ++i) {
    cin >> c;
    if (tolower(s1[i]) < tolower(c))
      END(-1)
    if (tolower(s1[i]) > tolower(c))
      END(1)
  }
  END(0)
}