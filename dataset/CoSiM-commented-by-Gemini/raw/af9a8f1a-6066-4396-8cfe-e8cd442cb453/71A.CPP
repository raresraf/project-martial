#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

void show(string s) {
  vector<char> vec;
  for (auto c : s)
    vec.push_back(c);
  if (vec.size() > 10)
    cout << s[0] << vec.size() - 2 << s[vec.size() - 1] << endl;
  else
    cout << s << endl;
}

int main() {
  string arr[500];
  int num;
  cin >> num;
  for (int i = 0; i < num; i++) {
    cin >> arr[i];
    // strcpy(arr[i],s);
  }
  for (int i = 0; i < num; i++) {
    show(arr[i]);
  }
  return 0;
}
