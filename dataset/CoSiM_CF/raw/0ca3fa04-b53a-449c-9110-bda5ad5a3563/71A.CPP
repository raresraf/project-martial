#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pi;

int main() {

  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n;
  cin >> n;

  while (n--) {
    string s;
    cin >> s;

    if (s.size() > 10) {
      cout << s.at(0) << s.size() - 2 << s.at(s.size() - 1) << "\n";
    } else {
      cout << s << endl;
    }
  }

  return 0;
}
