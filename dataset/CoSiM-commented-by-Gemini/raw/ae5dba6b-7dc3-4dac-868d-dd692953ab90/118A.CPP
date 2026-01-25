#include <bits/stdc++.h>
#include <iostream>
#include <math.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  // given vowels (a,e,i,o,u,y)
  for (int i = 0; i < str.length(); i++) {
    char c = tolower(str[i]);
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y')
      cout << "." << c;
  }
}
