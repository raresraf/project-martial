#include <bits/stdc++.h>

using namespace std;

int main() {

  string play0 = "0000000";
  string play1 = "1111111";

  string s;

  cin >> s;

  if (s.find(play1) != string::npos || s.find(play0) != string::npos)
    printf("YES\n");
  else
    printf("NO\n");
}