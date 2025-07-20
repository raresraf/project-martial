#include <bits/stdc++.h> // all c++ libraries

// defines
#define faster_io                                                              \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie()
#define range(i, a, b) for (int i = a; i <= b; i++)

using namespace std;

string str;
const string w = "hello";
int ind = 0;

int main() {

  faster_io;

  cin >> str;

  range(i, 0, str.size() - 1) {

    if (str[i] == w[ind]) {
      ind++;
    }
    if (ind == 5) {
      cout << "YES";
      return 0;
    }
  }
  cout << "NO";

  return 0;
}
