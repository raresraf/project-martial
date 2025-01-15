#include "bits/stdc++.h"

using namespace std;

int main(int argc, char const *argv[]) {
  string str;
  cin >> str;

  int inc = 0;
  string str2 = "hello";

  for (int i = 0; i < str.size(); i++) {
    if (str[i] == str2[inc]) {
      inc++;
    }
  }

  if (inc == 5)
    cout << "YES";
  else
    std::cout << "NO";
  return 0;
}
