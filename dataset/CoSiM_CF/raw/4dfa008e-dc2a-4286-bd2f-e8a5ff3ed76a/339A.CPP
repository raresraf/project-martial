#include "iostream"
#include <algorithm>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
using namespace std;
int main() {
  char s[100 + 5];
  cin >> s;
  vector<int> arr;
  int j = 0;
  for (int i = 0; i < strlen(s); ++i) {
    if (s[i] != '+') {
      if (s[i] == '1') {
        arr.push_back(1);
      } else if (s[i] == '2')
        arr.push_back(2);
      else if (s[i] == '3')
        arr.push_back(3);
    } else
      continue;
  }
  sort(arr.begin(), arr.end());
  for (int i = 0; i < arr.size(); ++i) {
    if (i != arr.size() - 1)
      cout << arr[i] << '+';
    else
      cout << arr[i] << endl;
  }
  // system("PAUSE");
  return 0;
}
