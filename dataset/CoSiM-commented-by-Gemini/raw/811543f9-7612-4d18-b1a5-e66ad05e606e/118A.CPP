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
  char word[100];
  int i = 0;
  vector<char> myvector;
  vector<char> myvector2;
  cin >> word;
  while (word[i] != '\0') {
    myvector.push_back(tolower(word[i]));
    i++;
  }
  for (i = 0; i < myvector.size(); i++) {
    if (myvector[i] == 'a' || myvector[i] == 'o' || myvector[i] == 'i' ||
        myvector[i] == 'y' || myvector[i] == 'e' || myvector[i] == 'u') {
      myvector.erase(myvector.begin() + i);
      i--;
    }
  }

  for (i = 0; i < myvector.size(); i++) {
    myvector2.push_back('.');
    myvector2.push_back(myvector[i]);
  }
  for (i = 0; i < myvector2.size(); i++) {
    cout << myvector2[i];
  }
  return 0;
}
