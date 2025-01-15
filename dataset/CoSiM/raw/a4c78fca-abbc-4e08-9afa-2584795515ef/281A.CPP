#include <algorithm>
#include <cstring>
#include <ctype.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

int main() {
  string word;
  vector<char> myvector;
  int i = 1;
  cin >> word;
  myvector.push_back(toupper(word[0]));

  while (word[i] != '\0') {
    myvector.push_back(word[i]);
    i++;
  }
  for (i = 0; i < myvector.size(); i++) {
    cout << myvector[i];
  }
  return 0;
}
