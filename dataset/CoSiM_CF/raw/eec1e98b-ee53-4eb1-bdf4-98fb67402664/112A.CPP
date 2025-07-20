#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string a, b;
  while (cin >> a >> b) {
    transform(a.begin(), a.end(), a.begin(), ::tolower);
    transform(b.begin(), b.end(), b.begin(), ::tolower);
    int ret = strcmp(a.c_str(), b.c_str());
    cout << (ret > 0 ? 1 : ret < 0 ? -1 : 0) << endl;
  }
  return 0;
}
