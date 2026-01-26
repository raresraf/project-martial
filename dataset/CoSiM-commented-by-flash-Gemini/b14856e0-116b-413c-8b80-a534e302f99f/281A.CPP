#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  string inp;
  while (cin >> inp) {
    inp[0] = toupper(inp[0]);
    cout << inp << endl;
  }
}
