#include <bits/stdc++.h>
#include <string>

using namespace std;

int main() {

  int i, j, n, t;

  string s;

  cin >> n >> t >> s;

  for (i = 1; i <= t; i++) { // t times loop

    for (j = 0; j < n; j++) { // index 0 to less than n

      if (s[j] == 'B' && s[j + 1] == 'G') {

        swap(s[j], s[j + 1]);
        j++; // after swapping two, goto next index
      }
    }
  }

  cout << s << endl;

  return 0;
}

// 666Lucifer666
