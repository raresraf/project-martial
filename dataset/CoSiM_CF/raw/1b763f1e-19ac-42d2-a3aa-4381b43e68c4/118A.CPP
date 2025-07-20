#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  string inp;
  char low;
  cin >> inp;
  string out = "";
  int j = 0, len;
  len = inp.size();
  for (int i = 0; i < len; i++) {
    low = tolower(inp[i]);
    if (low != 'a' && low != 'e' && low != 'i' && low != 'o' && low != 'u' &&
        low != 'y') {
      out = out + "." + low;
    }
  }
  cout << out << endl;
}
