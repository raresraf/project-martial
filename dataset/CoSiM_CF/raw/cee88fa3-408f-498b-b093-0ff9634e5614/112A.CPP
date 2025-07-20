#include "bits/stdc++.h"

using namespace std;

int main(int argc, char const *argv[]) {
  string str1, str2;
  cin >> str1 >> str2;

  int flag = 0;

  for (int i = 0; i < str1.size(); i++) {
    str1[i] = tolower(str1[i]);
    str2[i] = tolower(str2[i]);

    if (str1[i] > str2[i]) {
      flag = 1;
      break;
    } else if (str1[i] < str2[i]) {
      flag = -1;
      break;
    }
  }

  cout << flag;
  return 0;
}
