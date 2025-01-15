// 118A.cpp
// A. String Task
// Author: derekhh
// May 31, 2015

#include <iostream>
#include <string>
using namespace std;

bool isVowel[255];

int main() {
  isVowel['a'] = isVowel['e'] = isVowel['i'] = isVowel['o'] = isVowel['u'] =
      isVowel['y'] = true;
  string str, s1, s2, s3;
  cin >> str;
  for (int i = 0; i < str.size(); i++)
    if (!isVowel[tolower(str[i])])
      s1 += str[i];
  for (int i = 0; i < s1.size(); i++) {
    s2 += '.';
    s2 += s1[i];
  }
  for (int i = 0; i < s2.size(); i++) {
    if (isupper(s2[i]))
      s3 += tolower(s2[i]);
    else
      s3 += s2[i];
  }
  cout << s3 << endl;
  return 0;
}