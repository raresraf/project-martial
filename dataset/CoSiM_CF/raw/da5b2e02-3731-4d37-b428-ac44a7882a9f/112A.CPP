#include <bits/stdc++.h>
using namespace std;

int main() {
  string s1, s2;
  cin >> s1 >> s2;
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  int length;

  length = s1.length();
  /*
      for(int i=0; i<length; i++)
      {
          s1[i] = tolower(s1[i]);
          s2[i] = tolower(s2[i]);
      }
  */

  if (s1.compare(s2) == 0) {
    cout << "0" << endl;
  }

  else {
    for (int i = 0; i < length; i++) {
      if (s1[i] < s2[i]) {
        cout << "-1" << endl;
        break;
      } else if (s1[i] > s2[i]) {
        cout << "1" << endl;
        break;
      }
    }
  }
  return 0;
}