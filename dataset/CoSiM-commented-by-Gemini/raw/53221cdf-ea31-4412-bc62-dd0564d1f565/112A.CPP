#include <bits/stdc++.h>
#include <cstdio>
using namespace std;

int main() {
  char str[101], str2[101];
  int i, b;
  scanf("%s", str);
  scanf("%s", str2);
  for (i = 0; str[i]; i++) {
    if (tolower(str[i]) == tolower(str2[i])) {
      b = 0;
      continue;
    } else if (tolower(str[i]) != tolower(str2[i])) {
      if (tolower(str[i]) > tolower(str2[i])) {
        b = 1;
        break;
      } else if (tolower(str[i]) < tolower(str2[i])) {
        b = -1;
        break;
      }
    }
  }
  if (b == 0) {
    printf("0\n");
  } else if (b == 1) {
    printf("1\n");
  } else if (b == -1) {
    printf("-1\n");
  }
}

/*
#include <cstdio>
#include <iostream>
#include <string>

#include <cstring>
using namespace std;
char a1[110];
char a2[110];
char b1[110];
char b2[110];


int Cmp(char *pstr1,char *pstr2)
  {int i,t;
  for(i=0;pstr1[i]!='\0';i++)
  if(pstr1[i]>='A'&&pstr1[i]<='Z')
  *(pstr1+i)=*(pstr1+i)+32;
  for(i=0;pstr2[i]!='\0';i++)
  if(pstr2[i]>='A'&&pstr2[i]<='Z')
  *(pstr2+i)=*(pstr2+i)+32;
  t=0;
  for(i=0;i<strlen(pstr1);i++)
      {if(*(pstr1+i)>*(pstr2+i))
              {t=1;break;}
        if(*(pstr1+i)<*(pstr2+i))
              {t=-1;break;}
      if(*(pstr1+i)>*(pstr2+i))
      continue;

      }
        return t;
  }



int main()
  {int t;
  gets(a1);
  gets(a2);
  t=Cmp(a1,a2);

  if(t==1) cout<<1;
  if(t==-1)cout<<-1;
  if(t==0)cout<<0;
      cin.get();
          cout<<endl;

return 0;
}
*/
